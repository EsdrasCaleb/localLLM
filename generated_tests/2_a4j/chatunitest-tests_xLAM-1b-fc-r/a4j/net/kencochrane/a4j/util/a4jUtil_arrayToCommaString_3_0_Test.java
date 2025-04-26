package net.kencochrane.a4j.util;

import java.util.ArrayList;
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
import java.util.Properties;

public class a4jUtil_arrayToCommaString_3_0_Test {

    @Test
    public void arrayToCommaStringTest() {
        a4jUtil util = new a4jUtil();
        ArrayList<String> list = new ArrayList<>();
        list.add("Hello");
        list.add("World");
        assertEquals("Hello, World", util.arrayToCommaString(list));
        list = new ArrayList<>();
        list.add("Java");
        list.add("Test");
        assertEquals("Java, Test", util.arrayToCommaString(list));
        list = new ArrayList<>();
        assertEquals("", util.arrayToCommaString(list));
        list = null;
        assertEquals("", util.arrayToCommaString(list));
    }
}
