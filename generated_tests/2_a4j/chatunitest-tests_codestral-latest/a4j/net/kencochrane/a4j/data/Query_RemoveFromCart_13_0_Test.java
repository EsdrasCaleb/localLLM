package net.kencochrane.a4j.data;

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

public class Query_RemoveFromCart_13_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        query.serverURL = "http://xml.amazon.com/onca/xml3";
        query.associatesID = "test";
        query.searchType = "lite";
        query.type = "lite";
        query.page = "1";
        query.offer = "All";
        query.searchValues = new ArrayList<>();
    }

    @Test
    public void testRemoveFromCart() {
        String itemId = "17120277375791359165";
        String cartId = "CART";
        String hmac = "HMAC";
        String expectedHmac = "encodedHMAC";
        when(jawsUtil.encodeString(hmac)).thenReturn(expectedHmac);
        String result = query.RemoveFromCart(itemId, cartId, hmac);
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&f=xml&dev-t=DSB0XDDW1GQ3S&t=test&Item.17120277375791359165&CartId=CART&Hmac=encodedHMAC";
        assertEquals(expectedUrl, result);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
