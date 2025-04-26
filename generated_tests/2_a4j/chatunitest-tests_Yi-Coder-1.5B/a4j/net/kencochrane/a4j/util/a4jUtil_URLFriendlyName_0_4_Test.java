package net.kencochrane.a4j.util;

import java.util.Properties;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.params.provider.Arguments.arguments;
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

public class a4jUtil_URLFriendlyName_0_4_Test {

    private a4jUtil util;

    @BeforeEach
    void setUp() {
        util = new a4jUtil();
        MockitoAnnotations.openMocks(this);
    }

    @ParameterizedTest
    @MethodSource("provider")
    void testURLFriendlyName(String name, String expected) {
        String actual = util.URLFriendlyName(name);
        assertEquals(expected, actual);
    }

    private static Stream<Arguments> provider() {
        return Stream.of(arguments("Cool name", "cool-name"), arguments("Cool name with spaces", "cool-name-with-spaces"), arguments("Cool name with spaces and numbers", "cool-name-with-spaces-and-numbers"), arguments("Cool name with spaces and numbers and symbols", "cool-name-with-spaces-and-numbers-and-symbols"));
    }
}
