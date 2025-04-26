package net.kencochrane.a4j.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

class a4jUtil_arrayToCommaString_3_0_Test {

    @Test
    void testArrayToCommaString_nullList() {
        a4jUtil util = new a4jUtil();
        ArrayList<String> nullList = null;
        String result = util.arrayToCommaString(nullList);
        assertEquals("", result);
    }

    @Test
    void testArrayToCommaString_emptyList() {
        a4jUtil util = new a4jUtil();
        ArrayList<String> emptyList = new ArrayList<>();
        String result = util.arrayToCommaString(emptyList);
        assertEquals("", result);
    }

    @Test
    void testArrayToCommaString_singleElement() {
        a4jUtil util = new a4jUtil();
        ArrayList<String> list = new ArrayList<>(Arrays.asList("apple"));
        String result = util.arrayToCommaString(list);
        assertEquals("apple", result);
    }

    @Test
    void testArrayToCommaString_multipleElements() {
        a4jUtil util = new a4jUtil();
        ArrayList<String> list = new ArrayList<>(Arrays.asList("apple", "banana", "cherry"));
        String result = util.arrayToCommaString(list);
        assertEquals("apple, banana, cherry", result);
    }

    @Test
    void testArrayToCommaString_nullElement() {
        a4jUtil util = new a4jUtil();
        ArrayList<String> list = new ArrayList<>(Arrays.asList("apple", null, "cherry"));
        String result = util.arrayToCommaString(list);
        assertEquals("apple, cherry", result);
    }

    @Test
    void testArrayToCommaString_mixedElements() {
        a4jUtil util = new a4jUtil();
        ArrayList<String> list = new ArrayList<>(Arrays.asList("apple", "banana", null, "date"));
        String result = util.arrayToCommaString(list);
        assertEquals("apple, banana, date", result);
    }
}
