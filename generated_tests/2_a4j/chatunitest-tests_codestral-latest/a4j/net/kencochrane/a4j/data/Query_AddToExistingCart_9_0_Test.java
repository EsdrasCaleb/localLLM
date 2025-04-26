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

public class Query_AddToExistingCart_9_0_Test {

    @InjectMocks
    private Query query;

    @Mock
    private a4jUtil jawsUtil;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        query.serverURL = "http://xml.amazon.com/onca/xml3";
        query.associatesID = "testAssociatesID";
        query.token = "DSB0XDDW1GQ3S";
        query.searchType = "testSearchType";
        query.type = "testType";
        query.page = "testPage";
        query.offer = "testOffer";
        query.searchValues = new ArrayList<>();
    }

    @Test
    public void testAddToExistingCart() {
        String ASIN = "testASIN";
        String quantity = "1";
        String cartId = "testCartId";
        String hmac = "testHmac";
        String expectedEncodedHmac = "encodedHmac";
        when(jawsUtil.encodeString(hmac)).thenReturn(expectedEncodedHmac);
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociatesID&Asin.testASIN=1&CartId=testCartId&Hmac=encodedHmac";
        String actualUrl = query.AddToExistingCart(ASIN, quantity, cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
