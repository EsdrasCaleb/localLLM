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

public class Query_ClearCart_10_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Setting up protected fields using reflection
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "ASSOCIATES_ID");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
    }

    @Test
    void testClearCart() {
        String cartId = "CART_ID";
        String hmac = "HMAC_VALUE";
        String encodedHmac = "ENCODED_HMAC_VALUE";
        when(jawsUtil.encodeString(hmac)).thenReturn(encodedHmac);
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=ASSOCIATES_ID&CartId=CART_ID&Hmac=ENCODED_HMAC_VALUE";
        String actualUrl = query.ClearCart(cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
