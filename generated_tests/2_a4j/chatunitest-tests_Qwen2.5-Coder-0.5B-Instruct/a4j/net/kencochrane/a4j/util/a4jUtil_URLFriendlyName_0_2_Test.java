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

class a4jUtil_URLFriendlyName_0_2_Test {

    private a4jUtil instance;

    @BeforeEach
    public void setUp() {
        instance = spy(a4jUtil.class);
    }

    @Test
    public void testURLFriendlyName() {
        String input = "Hello World! This is a test.";
        String expectedOutput = "hello-world-this-is-a-test.";
        when(instance.URLFriendlyName(input)).thenReturn(expectedOutput);
        assertEquals(expectedOutput, instance.URLFriendlyName(input));
    }
}
