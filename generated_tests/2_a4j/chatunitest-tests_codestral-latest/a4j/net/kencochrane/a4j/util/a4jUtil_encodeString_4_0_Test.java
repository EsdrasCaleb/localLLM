package net.kencochrane.a4j.util;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_encodeString_4_0_Test {

    @InjectMocks
    private a4jUtil a4jUtil;

    @Test
    public void testEncodeString() throws UnsupportedEncodingException {
        String searchTerm = "test search term";
        String expectedEncodedTerm = URLEncoder.encode(searchTerm, "UTF-8");
        String actualEncodedTerm = a4jUtil.encodeString(searchTerm);
        assertEquals(expectedEncodedTerm, actualEncodedTerm);
    }
}
