package net.kencochrane.a4j.util;

import java.lang.reflect.Method;
import java.math.BigDecimal;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_getPrice_2_0_Test {

    @Test
    void testGetPriceValid() {
        a4jUtil util = new a4jUtil();
        assertEquals(new BigDecimal("12.34"), util.getPrice("12.34"));
        assertEquals(new BigDecimal("12.35"), util.getPrice("12.345"));
        assertEquals(new BigDecimal("12.34"), util.getPrice("12.346"));
        assertEquals(new BigDecimal("100.00"), util.getPrice("100"));
        assertEquals(new BigDecimal("0.00"), util.getPrice("0"));
    }

    @Test
    void testGetPriceInvalid() {
        a4jUtil util = new a4jUtil();
        assertEquals(new BigDecimal("0.00"), util.getPrice("abc"));
        assertEquals(new BigDecimal("0.00"), util.getPrice("12.ab"));
        assertEquals(new BigDecimal("0.00"), util.getPrice("12a.34"));
        assertEquals(new BigDecimal("0.00"), util.getPrice(null));
        assertEquals(new BigDecimal("0.00"), util.getPrice(""));
    }

    @Test
    void testStripString() throws Exception {
        a4jUtil util = new a4jUtil();
        Method stripStringMethod = a4jUtil.class.getDeclaredMethod("stripString", String.class, String.class);
        stripStringMethod.setAccessible(true);
        assertEquals("1234", stripStringMethod.invoke(util, ".0123456789", "1234"));
        assertEquals("1234", stripStringMethod.invoke(util, ".0123456789", "1234abc"));
        assertEquals("12.34", stripStringMethod.invoke(util, ".0123456789", "12.34"));
        assertEquals("", stripStringMethod.invoke(util, ".0123456789", "abc"));
        assertEquals("", stripStringMethod.invoke(util, ".0123456789", null));
        assertEquals("", stripStringMethod.invoke(util, ".0123456789", ""));
    }

    static class a4jUtil {

        public BigDecimal getPrice(String price) {
            try {
                BigDecimal bd = new BigDecimal(price);
                return bd.setScale(2, BigDecimal.ROUND_HALF_UP);
            } catch (NumberFormatException e) {
                return new BigDecimal("0.00");
            }
        }

        private String stripString(String allowedString, String price) {
            if (price == null || price.isEmpty()) {
                return "";
            }
            String result = "";
            for (int i = 0; i < price.length(); i++) {
                char c = price.charAt(i);
                if (allowedString.indexOf(c) != -1) {
                    result += c;
                }
            }
            return result;
        }
    }
}
