package net.kencochrane.a4j.util;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
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

@RunWith(MockitoJUnitRunner.class)
public class a4jUtil_URLFriendlyName_0_0_Test {

    @Mock
    private a4jUtil a4jUtil;

    @Test
    public void testURLFriendlyName() {
        String name = "Foo Bar";
        String expected = "foo-bar";
        String result = a4jUtil.URLFriendlyName(name);
        assertEquals(expected, result);
    }
}
