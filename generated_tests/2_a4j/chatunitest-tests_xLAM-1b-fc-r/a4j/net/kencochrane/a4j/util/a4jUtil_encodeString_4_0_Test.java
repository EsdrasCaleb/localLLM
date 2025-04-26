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

public class a4jUtil_encodeString_4_0_Test {

    private final a4jUtil a4jUtil = new a4jUtil();

    @Test
    public void testEncodeString() {
        String testString = "Hello, World!";
        String expectedOutput = "Hello%2C%20World%21";
        String actualOutput = a4jUtil.encodeString(testString);
        assertEquals(expectedOutput, actualOutput);
    }
}
