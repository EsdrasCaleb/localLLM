package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ListingProductDetails_toString_38_0_Test {

    @Test
    public void testToString() {
        ListingProductDetails listing = new ListingProductDetails();
        listing.setExchangeAsin("123456");
        listing.setExchangeAvailability("Available");
        listing.setExchangeConditionType("New");
        listing.setExchangeEndDate("2022-01-01");
        listing.setExchangeFeaturedCategory("Electronics");
        listing.setExchangeId("123456");
        listing.setExchangeOfferingType("Listing");
        listing.setExchangePrice("99.99");
        listing.setExchangeQuantity("100");
        listing.setExchangeQuantityAllocated("80");
        listing.setExchangeSellerCountry("USA");
        listing.setExchangeSellerId("123456");
        listing.setExchangeSellerNickname("John Doe");
        listing.setExchangeSellerRating("4.5");
        listing.setExchangeSellerState("California");
        listing.setExchangeStartDate("2022-01-01");
        listing.setExchangeStatus("Active");
        listing.setExchangeTitle("iPhone 13");
        String expected = " ----------- <br />\n" + "ASIN 123456<br />\n" + "Avail Available<br />\n" + "Condition Type New<br />\n" + "EndDate 2022-01-01<br />\n" + "Featured Cat Electronics<br />\n" + "Ex ID 123456<br />\n" + "Offer Type Listing<br />\n" + "Ex Price 99.99<br />\n" + "Ex Quant 100<br />\n" + "Quantity Allocated 80<br />\n" + "Seller Country USA<br />\n" + "Seller Id 123456<br />\n" + "Seller Nickname John Doe<br />\n" + "Seller Rating 4.5<br />\n" + "Seller State California<br />\n" + "Start date 2022-01-01<br />\n" + "Status Active<br />\n" + "Title iPhone 13<br />\n" + " ----------- <br />\n";
        assertEquals(expected, listing.toString());
    }
}
