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
    public void testGetPrice_ValidPrice() throws Exception {
        a4jUtil util = new a4jUtil();
        Method method = a4jUtil.class.getDeclaredMethod("getPrice", String.class);
        method.setAccessible(true);
        BigDecimal result = (BigDecimal) method.invoke(util, "123.45");
        assertEquals(new BigDecimal("123.45"), result);
    }

    @Test
    public void testGetPrice_ValidPriceWithExtraCharacters() throws Exception {
        a4jUtil util = new a4jUtil();
        Method method = a4jUtil.class.getDeclaredMethod("getPrice", String.class);
        method.setAccessible(true);
        BigDecimal result = (BigDecimal) method.invoke(util, "$123.45abc");
        assertEquals(new BigDecimal("123.45"), result);
    }

    @Test
    public void testGetPrice_InvalidPrice() throws Exception {
        a4jUtil util = new a4jUtil();
        Method method = a4jUtil.class.getDeclaredMethod("getPrice", String.class);
        method.setAccessible(true);
        BigDecimal result = (BigDecimal) method.invoke(util, "invalidPrice");
        assertEquals(BigDecimal.ZERO, result);
    }

    @Test
    public void testGetPrice_EmptyString() throws Exception {
        a4jUtil util = new a4jUtil();
        Method method = a4jUtil.class.getDeclaredMethod("getPrice", String.class);
        method.setAccessible(true);
        BigDecimal result = (BigDecimal) method.invoke(util, "");
        assertEquals(BigDecimal.ZERO, result);
    }

    @Test
    public void testGetPrice_NullInput() throws Exception {
        a4jUtil util = new a4jUtil();
        Method method = a4jUtil.class.getDeclaredMethod("getPrice", String.class);
        method.setAccessible(true);
        BigDecimal result = (BigDecimal) method.invoke(util, (String) null);
        assertEquals(BigDecimal.ZERO, result);
    }
}
