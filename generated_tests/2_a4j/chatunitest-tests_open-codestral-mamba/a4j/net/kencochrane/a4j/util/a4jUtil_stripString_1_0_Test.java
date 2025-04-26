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
        // Test case 1: allowedChars = "abc", string = "defg"
        String allowedChars1 = "abc";
        String string1 = "defg";
        String expected1 = "def";
        assertEquals(expected1, util.stripString(allowedChars1, string1));
        // Test case 2: allowedChars = "123", string = "4567"
        String allowedChars2 = "123";
        String string2 = "4567";
        String expected2 = "456";
        assertEquals(expected2, util.stripString(allowedChars2, string2));
        // Test case 3: allowedChars = "abc", string = "abc"
        String allowedChars3 = "abc";
        String string3 = "abc";
        String expected3 = "abc";
        assertEquals(expected3, util.stripString(allowedChars3, string3));
        // Test case 4: allowedChars = "abc", string = ""
        String allowedChars4 = "abc";
        String string4 = "";
        String expected4 = "";
        assertEquals(expected4, util.stripString(allowedChars4, string4));
    }
}
