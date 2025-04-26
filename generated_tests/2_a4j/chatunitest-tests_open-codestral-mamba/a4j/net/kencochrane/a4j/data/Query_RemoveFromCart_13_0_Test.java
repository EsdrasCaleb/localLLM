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

public class Query_RemoveFromCart_13_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testRemoveFromCart() {
        String itemId = "123456789";
        String cartId = "987654321";
        String hmac = "hmac123";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&CartId=987654321&Hmac=encodedHmac&Item.123456789&dev-t=DSB0XDDW1GQ3S&t=test&f=xml&type=lite";
        when(jawsUtil.encodeString("hmac123")).thenReturn("encodedHmac");
        String actualUrl = query.RemoveFromCart(itemId, cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
    }
}
