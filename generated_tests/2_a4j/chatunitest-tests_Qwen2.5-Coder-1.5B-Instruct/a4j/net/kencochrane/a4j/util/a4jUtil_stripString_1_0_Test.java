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

public class a4jUtil_stripString_1_0_Test {

    private a4jUtil a4jUtil;

    @BeforeEach
    public void setUp() {
        a4jUtil = new a4jUtil();
    }

    @Test
    public void testStripStringWithAllowedCharsOnly() {
        assertEquals("abc", a4jUtil.stripString("abc", "abcdef"));
    }

    @Test
    public void testStripStringWithEmptyAllowedChars() {
        assertEquals("", a4jUtil.stripString("", "abcdef"));
    }

    @Test
    public void testStripStringWithNonMatchingCharacters() {
        assertEquals("b", a4jUtil.stripString("bc", "abcdef"));
    }

    @Test
    public void testStripStringWithAllMatchingCharacters() {
        assertEquals("abcdef", a4jUtil.stripString("abcdef", "abcdef"));
    }
}
