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

public class Query_GetItemsFromCart_11_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        query.serverURL = "http://xml.amazon.com/onca/xml3";
        query.associatesID = "testAssociateID";
    }

    @Test
    public void testGetItemsFromCart() {
        String cartId = "testCartId";
        String hmac = "testHmac";
        String expectedEncodedHmac = "encodedHmac";
        when(jawsUtil.encodeString(hmac)).thenReturn(expectedEncodedHmac);
        String result = query.GetItemsFromCart(cartId, hmac);
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=get&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociateID&CartId=testCartId&Hmac=encodedHmac";
        assertEquals(expectedUrl, result);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
