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

    @Test
    public void testStripString() {
        a4jUtil util = new a4jUtil();
        String result = util.stripString("abcd", "abcdefgh");
        assertEquals("", result);
        result = util.stripString("abcd", "abcde");
        assertEquals("", result);
        result = util.stripString("abcd", "abcdefghi");
        assertEquals("ghi", result);
        result = util.stripString("abcd", "aabbccdd");
        assertEquals("", result);
        result = util.stripString("abcd", "aabbcc");
        assertEquals("", result);
        result = util.stripString("abcd", "abcd");
        assertEquals("", result);
        result = util.stripString("abcd", "");
        assertEquals("", result);
        result = util.stripString("abcd", "abcdefgh");
        assertEquals("gh", result);
    }
}
