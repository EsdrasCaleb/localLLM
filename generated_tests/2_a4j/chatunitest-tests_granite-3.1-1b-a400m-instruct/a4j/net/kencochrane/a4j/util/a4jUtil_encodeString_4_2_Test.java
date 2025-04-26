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

public class a4jUtil_encodeString_4_2_Test {

    @Test
    public void testEncodeString() {
        a4jUtil util = new a4jUtil();
        String expected = "a4j:encode";
        String actual = util.encodeString("a4j");
        assertEquals(expected, actual);
    }

    @Test
    public void testEncodeStringWithInvalidScheme() {
        a4jUtil util = new a4jUtil();
        String expected = "a4j:encode";
        String actual = util.encodeString("invalid");
        assertEquals(expected, actual);
    }
}
