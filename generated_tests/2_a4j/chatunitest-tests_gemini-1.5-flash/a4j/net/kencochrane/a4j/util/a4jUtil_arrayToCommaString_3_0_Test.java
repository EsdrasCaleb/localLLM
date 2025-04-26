package net.kencochrane.a4j.util;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
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

    @ParameterizedTest
    @MethodSource("provideTestCases")
    void testArrayToCommaString(ArrayList<String> inputList, String expectedOutput) {
        a4jUtil util = new a4jUtil();
        String result = util.arrayToCommaString(inputList);
        assertEquals(expectedOutput, result);
    }

    static Stream<Arguments> provideTestCases() {
        return Stream.of(Arguments.of(null, ""), Arguments.of(new ArrayList<>(), ""), Arguments.of(new ArrayList<>(Arrays.asList("apple")), "apple"), Arguments.of(new ArrayList<>(Arrays.asList("apple", "banana", "cherry")), "apple, banana, cherry"), Arguments.of(new ArrayList<>(Arrays.asList("apple", null, "cherry")), "apple, cherry"), Arguments.of(new ArrayList<>(Arrays.asList(null, null, null)), ""), Arguments.of(new ArrayList<>(Arrays.asList("apple", "banana", null)), "apple, banana"));
    }

    @Test
    void testArrayToCommaString_empty() {
        a4jUtil util = new a4jUtil();
        assertEquals("", util.arrayToCommaString(new ArrayList<>()));
    }

    @Test
    void testArrayToCommaString_null() {
        a4jUtil util = new a4jUtil();
        assertEquals("", util.arrayToCommaString(null));
    }

    @Test
    void testArrayToCommaString_singleElement() {
        a4jUtil util = new a4jUtil();
        assertEquals("apple", util.arrayToCommaString(new ArrayList<>(Arrays.asList("apple"))));
    }

    @Test
    void testArrayToCommaString_multipleElements() {
        a4jUtil util = new a4jUtil();
        assertEquals("apple, banana, cherry", util.arrayToCommaString(new ArrayList<>(Arrays.asList("apple", "banana", "cherry"))));
    }

    @Test
    void testArrayToCommaString_nullElements() {
        a4jUtil util = new a4jUtil();
        assertEquals("apple, cherry", util.arrayToCommaString(new ArrayList<>(Arrays.asList("apple", null, "cherry"))));
    }
}
