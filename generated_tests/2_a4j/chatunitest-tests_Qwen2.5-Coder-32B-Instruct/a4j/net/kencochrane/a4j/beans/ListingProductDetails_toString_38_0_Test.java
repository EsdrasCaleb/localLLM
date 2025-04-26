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

    private ListingProductDetails listingProductDetails;

    @BeforeEach
    public void setUp() {
        listingProductDetails = new ListingProductDetails();
    }

    private void setField(String fieldName, Object value) throws Exception {
        Field field = ListingProductDetails.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(listingProductDetails, value);
    }

    @Test
    public void testToString() throws Exception {
        // Setting all fields to non-null values
        setField("exchangeId", "12345");
        setField("listingId", "67890");
        setField("exchangeTitle", "Test Product");
        setField("exchangePrice", "19.99");
        setField("exchangeAsin", "B08N5WRWNW");
        setField("exchangeEndDate", "2023-12-31");
        setField("exchangeOfferingType", "New");
        setField("exchangeSellerId", "SELLER123");
        setField("exchangeSellerNickname", "SellerNick");
        setField("exchangeStartDate", "2023-01-01");
        setField("exchangeStatus", "Active");
        setField("exchangeQuantity", "10");
        setField("exchangeQuantityAllocated", "5");
        setField("exchangeFeaturedCategory", "Electronics");
        setField("exchangeConditionType", "New");
        setField("exchangeAvailability", "In Stock");
        setField("exchangeSellerState", "CA");
        setField("exchangeSellerCountry", "USA");
        setField("exchangeSellerRating", "4.5");
        String expectedOutput = " ----------- <br />\n" + "ASIN B08N5WRWNW<br />\n" + "Avail In Stock<br />\n" + "Condition Type New<br />\n" + "EndDate 2023-12-31<br />\n" + "Featured Cat Electronics<br />\n" + "Ex ID 12345<br />\n" + "Offer Type New<br />\n" + "Ex Price 19.99<br />\n" + "Ex Quant 10<br />\n" + "Quantity Allocated 5<br />\n" + "Seller Country USA<br />\n" + "Seller Id SELLER123<br />\n" + "Seller Nickname SellerNick<br />\n" + "Seller Rating 4.5<br />\n" + "Seller State CA<br />\n" + "Start date 2023-01-01<br />\n" + "Status Active<br />\n" + "Title Test Product<br />\n" + " ----------- <br />\n";
        String actualOutput = listingProductDetails.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToStringWithNullValues() throws Exception {
        // Setting all fields to null values
        setField("exchangeId", null);
        setField("listingId", null);
        setField("exchangeTitle", null);
        setField("exchangePrice", null);
        setField("exchangeAsin", null);
        setField("exchangeEndDate", null);
        setField("exchangeOfferingType", null);
        setField("exchangeSellerId", null);
        setField("exchangeSellerNickname", null);
        setField("exchangeStartDate", null);
        setField("exchangeStatus", null);
        setField("exchangeQuantity", null);
        setField("exchangeQuantityAllocated", null);
        setField("exchangeFeaturedCategory", null);
        setField("exchangeConditionType", null);
        setField("exchangeAvailability", null);
        setField("exchangeSellerState", null);
        setField("exchangeSellerCountry", null);
        setField("exchangeSellerRating", null);
    }
}
