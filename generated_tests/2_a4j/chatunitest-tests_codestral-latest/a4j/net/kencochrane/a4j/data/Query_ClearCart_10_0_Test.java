package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
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
import java.util.Properties;

class Query_ClearCart_10_0_Test {

    @InjectMocks
    private Query query;

    @Mock
    private a4jUtil jawsUtil;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        MockitoAnnotations.openMocks(this);
        Field tokenField = Query.class.getDeclaredField("token");
        tokenField.setAccessible(true);
        tokenField.set(query, "DSB0XDDW1GQ3S");
    }

    @Test
    void testClearCart() {
        String cartId = "12345";
        String hmac = "testHmac";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=&CartId=12345&Hmac=encodedHmac";
        when(jawsUtil.encodeString(hmac)).thenReturn("encodedHmac");
        String result = query.ClearCart(cartId, hmac);
        assertEquals(expectedUrl, result);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
