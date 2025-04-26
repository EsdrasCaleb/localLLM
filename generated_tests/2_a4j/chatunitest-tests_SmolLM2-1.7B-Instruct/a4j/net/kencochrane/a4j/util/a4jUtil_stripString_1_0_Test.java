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
    public void testStripString_InvalidAllowedChars() {
        a4jUtil util = new a4jUtil();
        String allowedChars = "xyz";
        String string = "Hello World";
        String expected = "";
        String actual = util.stripString(allowedChars, string);
        assertEquals(expected, actual);
    }
}
