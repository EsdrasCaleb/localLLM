package net.kencochrane.a4j.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_getPrice_2_0_Test {

    @InjectMocks
    private a4jUtil a4jUtil;

    @Test
    public void testGetPrice_ValidString_ReturnsCorrectPrice() {
        assertEquals(new BigDecimal("12.34"), a4jUtil.getPrice("12.34"));
    }

    @Test
    public void testGetPrice_ValidStringWithLeadingZeroes_ReturnsCorrectPrice() {
        assertEquals(new BigDecimal("012.34"), a4jUtil.getPrice("012.34"));
    }

    @Test
    public void testGetPrice_ValidStringWithSpace_ReturnsCorrectPrice() {
        assertEquals(new BigDecimal("12.34"), a4jUtil.getPrice(" 12.34 "));
    }
}
