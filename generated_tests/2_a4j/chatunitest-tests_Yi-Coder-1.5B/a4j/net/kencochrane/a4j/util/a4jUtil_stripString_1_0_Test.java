package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_stripString_1_0_Test {

    // Test class
    @Test
    public void testStripString() {
        a4jUtil a4jUtil = new a4jUtil();
        assertEquals("ab", a4jUtil.stripString("abc", "ab"));
        assertEquals("ab", a4jUtil.stripString("abc", "a"));
        assertEquals("", a4jUtil.stripString("abc", ""));
        assertEquals("", a4jUtil.stripString("abc", null));
        assertEquals("", a4jUtil.stripString("", "abc"));
        assertEquals("", a4jUtil.stripString(null, "abc"));
        assertEquals("", a4jUtil.stripString(null, null));
        assertEquals("", a4jUtil.stripString("", ""));
        assertEquals("", a4jUtil.stripString("", null));
        assertEquals("", a4jUtil.stripString(null, ""));
    }
}
