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
public class a4jUtil_dencodeString_5_0_Test {

    @Test
    public void testDencodeString_UTF8() {
        String input = "Hello%20World";
        String expected = "Hello World";
        a4jUtil instance = new a4jUtil();
        String actual = instance.dencodeString(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testDencodeString_Fallback() {
        String input = "Hello%20World";
        String expected = "Hello World";
        a4jUtil instance = new a4jUtil();
        String actual = instance.dencodeString(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testDencodeString_NullInput() {
        String input = null;
        String expected = null;
        a4jUtil instance = new a4jUtil();
        String actual = instance.dencodeString(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testDencodeString_EmptyInput() {
        String input = "";
        String expected = "";
        a4jUtil instance = new a4jUtil();
        String actual = instance.dencodeString(input);
        assertEquals(expected, actual);
    }
}
