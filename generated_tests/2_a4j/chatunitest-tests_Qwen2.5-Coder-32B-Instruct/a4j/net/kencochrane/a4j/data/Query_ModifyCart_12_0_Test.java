package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
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

public class Query_ModifyCart_12_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        query = new Query();
        // Set the mock jawsUtil into the query object using reflection
        Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
        jawsUtilField.setAccessible(true);
        jawsUtilField.set(query, jawsUtil);
        // Set the serverURL and associatesID using reflection
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "ASSOCIATES_ID");
    }

    @Test
    public void testModifyCart() {
        String itemId = "12345";
        String quantity = "2";
        String cartId = "CART_ID";
        String hmac = "HMAC_VALUE";
        String encodedHmac = "ENCODED_HMAC_VALUE";
        when(jawsUtil.encodeString(hmac)).thenReturn(encodedHmac);
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=ASSOCIATES_ID&Item.12345=2&CartId=CART_ID&Hmac=ENCODED_HMAC_VALUE";
        String result = query.ModifyCart(itemId, quantity, cartId, hmac);
        assertEquals(expectedUrl, result);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
