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

public class a4jUtil_arrayToCommaString_3_1_Test {

    @Test
    public void testArrayToCommaString() {
        a4jUtil util = new a4jUtil();
        // Test with null input
        ArrayList<String> nullInput = null;
        assertEquals("", util.arrayToCommaString(nullInput));
        // Test with empty input
        ArrayList<String> emptyInput = new ArrayList<>();
        assertEquals("", util.arrayToCommaString(emptyInput));
        // Test with input containing one element
        ArrayList<String> oneElementInput = new ArrayList<>();
        oneElementInput.add("element");
        assertEquals("element", util.arrayToCommaString(oneElementInput));
        // Test with input containing multiple elements
        ArrayList<String> multipleElementsInput = new ArrayList<>();
        multipleElementsInput.add("element1");
        multipleElementsInput.add("element2");
        multipleElementsInput.add("element3");
        assertEquals("element1, element2, element3", util.arrayToCommaString(multipleElementsInput));
    }
}
