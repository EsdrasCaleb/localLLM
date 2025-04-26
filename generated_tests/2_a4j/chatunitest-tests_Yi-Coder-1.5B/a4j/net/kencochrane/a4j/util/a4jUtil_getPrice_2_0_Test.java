package net.kencochrane.a4j.util;

import java.math.BigDecimal;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_getPrice_2_0_Test {

    @Test
    public void testGetPrice() {
        String price = "1234.56";
        BigDecimal expected = new BigDecimal(1234.56);
        BigDecimal actual = new a4jUtil().getPrice(price);
        Assertions.assertEquals(expected, actual);
    }
}
