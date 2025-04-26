package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ListingProductDetails_toString_38_0_Test {

    @Test
    public void testToString() throws Exception {
        ListingProductDetails product = new ListingProductDetails();
        product.setExchangeId("12345");
        product.setExchangeTitle("Sample Title");
        product.setExchangePrice("99.99");
        product.setExchangeAsin("B0789ABCDEF");
        product.setExchangeEndDate("2023-10-31");
        product.setExchangeOfferingType("Fixed Price");
        product.setExchangeSellerId("seller123");
        product.setExchangeSellerNickname("SellerName");
        product.setExchangeStartDate("2023-09-01");
        product.setExchangeStatus("Active");
        product.setExchangeQuantity("10");
        product.setExchangeQuantityAllocated("5");
        product.setExchangeFeaturedCategory("Electronics");
        product.setExchangeConditionType("New");
        product.setExchangeAvailability("In Stock");
        product.setExchangeSellerState("California");
        product.setExchangeSellerCountry("USA");
        product.setExchangeSellerRating("4.5/5");
        String expectedOutput = " ----------- <br />\n" + "ASIN B0789ABCDEF<br />\n" + "Avail In Stock<br />\n" + "Condition Type New<br />\n" + "EndDate 2023-10-31<br />\n" + "Featured Cat Electronics<br />\n" + "Ex ID 12345<br />\n" + "Offer Type Fixed Price<br />\n" + "Ex Price $99.99<br />\n" + "Ex Quant 10<br />\n" + "Quantity Allocated 5<br />\n" + "Seller Country USA<br />\n" + "Seller Id seller123<br />\n" + "Seller Nickname SellerName<br />\n" + "Seller Rating 4.5/5<br />\n" + "Seller State California<br />\n" + "Start date 2023-09-01<br />\n" + "Status Active<br />\n" + "Title Sample Title<br />\n" + " ----------- <br />";
        assertEquals(expectedOutput, product.toString());
    }
}
