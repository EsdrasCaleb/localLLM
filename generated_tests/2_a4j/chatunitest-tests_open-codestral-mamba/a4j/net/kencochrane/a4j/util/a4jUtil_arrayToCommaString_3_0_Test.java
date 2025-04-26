package net.kencochrane.a4j.util;

import java.util.ArrayList;
import java.util.Arrays;
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
    public void testArrayToCommaString() {
        a4jUtil a4jUtil = Mockito.spy(new a4jUtil());
        ArrayList<String> list = new ArrayList<>(Arrays.asList("one", "two", "three"));
        String expected = "one, two, three";
        String actual = a4jUtil.arrayToCommaString(list);
        assertEquals(expected, actual);
    }

    @Test
    public void testArrayToCommaStringWithNullValues() {
        a4jUtil a4jUtil = Mockito.spy(new a4jUtil());
        ArrayList<String> list = new ArrayList<>(Arrays.asList("one", null, "three"));
        String expected = "one, three";
        String actual = a4jUtil.arrayToCommaString(list);
        assertEquals(expected, actual);
    }

    @Test
    public void testArrayToCommaStringWithEmptyList() {
        a4jUtil a4jUtil = Mockito.spy(new a4jUtil());
        ArrayList<String> list = new ArrayList<>();
        String expected = "";
        String actual = a4jUtil.arrayToCommaString(list);
        assertEquals(expected, actual);
    }

    @Test
    public void testArrayToCommaStringWithNullList() {
        a4jUtil a4jUtil = Mockito.spy(new a4jUtil());
        ArrayList<String> list = null;
        String expected = "";
        String actual = a4jUtil.arrayToCommaString(list);
        assertEquals(expected, actual);
    }
}
