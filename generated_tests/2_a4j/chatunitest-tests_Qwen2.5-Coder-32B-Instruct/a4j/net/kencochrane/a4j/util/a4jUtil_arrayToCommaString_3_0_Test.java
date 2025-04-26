package net.kencochrane.a4j.util;

import java.lang.reflect.Method;
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

    private a4jUtil util;

    @BeforeEach
    public void setUp() {
        util = new a4jUtil();
    }

    @Test
    public void testArrayToCommaString_withNullList() throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("arrayToCommaString", ArrayList.class);
        method.setAccessible(true);
        String result = (String) method.invoke(util, (ArrayList) null);
        assertEquals("", result);
    }

    @Test
    public void testArrayToCommaString_withEmptyList() throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("arrayToCommaString", ArrayList.class);
        method.setAccessible(true);
        ArrayList<Object> list = new ArrayList<>();
        String result = (String) method.invoke(util, list);
        assertEquals("", result);
    }

    @Test
    public void testArrayToCommaString_withSingleElement() throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("arrayToCommaString", ArrayList.class);
        method.setAccessible(true);
        ArrayList<Object> list = new ArrayList<>();
        list.add("Hello");
        String result = (String) method.invoke(util, list);
        assertEquals("Hello", result);
    }

    @Test
    public void testArrayToCommaString_withMultipleElements() throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("arrayToCommaString", ArrayList.class);
        method.setAccessible(true);
        ArrayList<Object> list = new ArrayList<>();
        list.add("Hello");
        list.add("World");
        list.add("Test");
        String result = (String) method.invoke(util, list);
        assertEquals("Hello, World, Test", result);
    }

    @Test
    public void testArrayToCommaString_withNullElements() throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("arrayToCommaString", ArrayList.class);
        method.setAccessible(true);
        ArrayList<Object> list = new ArrayList<>();
        list.add("Hello");
        list.add(null);
        list.add("World");
        list.add(null);
        list.add("Test");
        String result = (String) method.invoke(util, list);
        assertEquals("Hello, World, Test", result);
    }
}
