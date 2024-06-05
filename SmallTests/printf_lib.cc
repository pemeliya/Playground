#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
extern "C"
{

int __sprintf_chk(char* s, int flag, size_t slen, const char* format, ...)
{
  va_list arg;
  int done;

  va_start (arg, format);

  if (slen > (size_t) INT_MAX)
    done = vsprintf (s, format, arg);
  else
    {
      done = vsnprintf (s, slen, format, arg);
    //   if (done >= 0 && (size_t) done >= slen)
	//         __chk_fail ();
    }

  va_end (arg);

  return done;
}
}