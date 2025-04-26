package net.kencochrane.a4j.util;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
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
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_encodeString_4_0_Test {

    @Test
    public void testEncodeString_utf8Supported() throws UnsupportedEncodingException {
        a4jUtil util = new a4jUtil();
        String input = "Hello, world!";
        String expected = URLEncoder.encode(input, StandardCharsets.UTF_8.toString());
        String actual = util.encodeString(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testEncodeString_emptyInput() {
        a4jUtil util = new a4jUtil();
        String input = "";
        String actual = util.encodeString(input);
        assertEquals("", actual);
    }
}
