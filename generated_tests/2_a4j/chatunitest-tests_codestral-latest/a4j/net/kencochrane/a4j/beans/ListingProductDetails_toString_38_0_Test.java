package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ListingProductDetails_toString_38_0_Test {

    @InjectMocks
    private ListingProductDetails listingProductDetails;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the private fields using reflection
        Field[] fields = ListingProductDetails.class.getDeclaredFields();
        for (Field field : fields) {
            field.setAccessible(true);
            field.set(listingProductDetails, "TestValue");
        }
    }

    @Test
    public void testToString() {
        String expected = " ----------- <br />\n" + "ASIN TestValue<br />\n" + "Avail TestValue<br />\n" + "Condition Type TestValue<br />\n" + "EndDate TestValue<br />\n" + "Featured Cat TestValue<br />\n" + "Ex ID TestValue<br />\n" + "Offer Type TestValue<br />\n" + "Ex Price TestValue<br />\n" + "Ex Quant TestValue<br />\n" + "Quantity Allocated TestValue<br />\n" + "Seller Country TestValue<br />\n" + "Seller Id TestValue<br />\n" + "Seller Nickname TestValue<br />\n" + "Seller Rating TestValue<br />\n" + "Seller State TestValue<br />\n" + "Start date TestValue<br />\n" + "Status TestValue<br />\n" + "Title TestValue<br />\n" + " ----------- <br />\n";
        String actual = listingProductDetails.toString();
        assertEquals(expected, actual);
    }
}
