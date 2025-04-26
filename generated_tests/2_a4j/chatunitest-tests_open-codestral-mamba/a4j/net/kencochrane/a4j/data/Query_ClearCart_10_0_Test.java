package net.kencochrane.a4j.data;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;

public class Query_ClearCart_10_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testClearCart() {
        String cartId = "12345";
        String hmac = "testHmac";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=&CartId=12345&Hmac=encodedHmac";
        when(jawsUtil.encodeString(hmac)).thenReturn("encodedHmac");
        String actualUrl = query.ClearCart(cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
