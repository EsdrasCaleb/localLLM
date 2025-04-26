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

    private final a4jUtil util = new a4jUtil();

    @Test
    public void testArrayToCommaString_NullList() {
        String result = util.arrayToCommaString(null);
        assertEquals("", result);
    }

    @Test
    public void testArrayToCommaString_EmptyList() {
        ArrayList<String> list = new ArrayList<>();
        String result = util.arrayToCommaString(list);
        assertEquals("", result);
    }

    @Test
    public void testArrayToCommaString_SingleElement() {
        ArrayList<String> list = new ArrayList<>();
        list.add("Hello");
        String result = util.arrayToCommaString(list);
        assertEquals("Hello", result);
    }

    @Test
    public void testArrayToCommaString_MultipleElements() {
        ArrayList<String> list = new ArrayList<>();
        list.add("Hello");
        list.add("World");
        String result = util.arrayToCommaString(list);
        assertEquals("Hello, World", result);
    }

    @Test
    public void testArrayToCommaString_WithNullElements() {
        ArrayList<String> list = new ArrayList<>();
        list.add("Hello");
        list.add(null);
        list.add("World");
        String result = util.arrayToCommaString(list);
        assertEquals("Hello, World", result);
    }

    @Test
    public void testArrayToCommaString_AllNullElements() {
        ArrayList<String> list = new ArrayList<>();
        list.add(null);
        list.add(null);
        String result = util.arrayToCommaString(list);
        assertEquals("", result);
    }
}
