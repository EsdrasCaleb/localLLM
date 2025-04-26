package net.kencochrane.a4j.beans;

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
        listingProductDetails.setExchangeAsin("B000123456");
        listingProductDetails.setExchangeAvailability("In Stock");
        listingProductDetails.setExchangeConditionType("New");
        listingProductDetails.setExchangeEndDate("2023-12-31");
        listingProductDetails.setExchangeFeaturedCategory("Electronics");
        listingProductDetails.setExchangeId("EX12345");
        listingProductDetails.setExchangeOfferingType("Buy Now");
        listingProductDetails.setExchangePrice("$99.99");
        listingProductDetails.setExchangeQuantity("10");
        listingProductDetails.setExchangeQuantityAllocated("2");
        listingProductDetails.setExchangeSellerCountry("USA");
        listingProductDetails.setExchangeSellerId("SELLER123");
        listingProductDetails.setExchangeSellerNickname("BestSeller");
        listingProductDetails.setExchangeSellerRating("4.5");
        listingProductDetails.setExchangeSellerState("CA");
        listingProductDetails.setExchangeStartDate("2023-01-01");
        listingProductDetails.setExchangeStatus("Active");
        listingProductDetails.setExchangeTitle("Amazing Product");
    }

    @Test
    public void testToString() {
        String expectedOutput = " ----------- <br />\n" + "ASIN B000123456<br />\n" + "Avail In Stock<br />\n" + "Condition Type New<br />\n" + "EndDate 2023-12-31<br />\n" + "Featured Cat Electronics<br />\n" + "Ex ID EX12345<br />\n" + "Offer Type Buy Now<br />\n" + "Ex Price $99.99<br />\n" + "Ex Quant 10<br />\n" + "Quantity Allocated 2<br />\n" + "Seller Country USA<br />\n" + "Seller Id SELLER123<br />\n" + "Seller Nickname BestSeller<br />\n" + "Seller Rating 4.5<br />\n" + "Seller State CA<br />\n" + "Start date 2023-01-01<br />\n" + "Status Active<br />\n" + "Title Amazing Product<br />\n" + " ----------- <br />\n";
        String actualOutput = listingProductDetails.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
