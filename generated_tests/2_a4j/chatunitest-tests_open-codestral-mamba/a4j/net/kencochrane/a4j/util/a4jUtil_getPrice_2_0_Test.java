package net.kencochrane.a4j.util;

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
    public void testGetPrice() {
        a4jUtil a4jUtil = Mockito.mock(a4jUtil.class);
        Mockito.when(a4jUtil.getPrice("123.45")).thenReturn(new BigDecimal("123.45"));
        Mockito.when(a4jUtil.getPrice("abc")).thenReturn(new BigDecimal("0.00"));
        assertEquals(new BigDecimal("123.45"), a4jUtil.getPrice("123.45"));
        assertEquals(new BigDecimal("0.00"), a4jUtil.getPrice("abc"));
    }
}
