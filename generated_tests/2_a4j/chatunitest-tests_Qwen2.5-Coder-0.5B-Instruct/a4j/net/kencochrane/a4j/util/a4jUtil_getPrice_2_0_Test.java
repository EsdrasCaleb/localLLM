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

class a4jUtil_getPrice_2_0_Test {

    @Test
    public void testGetPrice() throws Exception {
        a4jUtil util = new a4jUtil();
        String price = "123.456";
        BigDecimal expectedPrice = new BigDecimal("123.46");
        BigDecimal actualPrice = util.getPrice(price);
        assertEquals(expectedPrice, actualPrice);
    }
}
