package net.kencochrane.a4j.util;

import java.lang.reflect.Field;
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

    @Test
    void testStripString_emptyAllowedChars() {
        a4jUtil util = new a4jUtil();
        assertEquals("", util.stripString("", "test"));
    }

    @Test
    void testStripString_emptyString() {
        a4jUtil util = new a4jUtil();
        assertEquals("", util.stripString("abc", ""));
    }

    @Test
    void testStripString_allCharsAllowed() {
        a4jUtil util = new a4jUtil();
        assertEquals("test", util.stripString("test", "test"));
    }

    @Test
    void testStripString_someCharsAllowed() {
        a4jUtil util = new a4jUtil();
        assertEquals("te", util.stripString("te", "test"));
    }

    @Test
    void testStripString_noCharsAllowed() {
        a4jUtil util = new a4jUtil();
        assertEquals("", util.stripString("abc", "test"));
    }

    @Test
    void testStripString_specialChars() {
        a4jUtil util = new a4jUtil();
        assertEquals("!$", util.stripString("!$", "test!$"));
    }

    @Test
    void testStripString_mixedCase() {
        a4jUtil util = new a4jUtil();
        assertEquals("TeSt", util.stripString("TeSt", "tEst"));
    }

    @Test
    void testStripString_duplicateChars() {
        a4jUtil util = new a4jUtil();
        assertEquals("aa", util.stripString("aa", "aaaa"));
    }

    @Test
    void testStripString_nullInput() {
        a4jUtil util = new a4jUtil();
        assertThrows(NullPointerException.class, () -> util.stripString(null, "test"));
        assertThrows(NullPointerException.class, () -> util.stripString("test", null));
    }
}
