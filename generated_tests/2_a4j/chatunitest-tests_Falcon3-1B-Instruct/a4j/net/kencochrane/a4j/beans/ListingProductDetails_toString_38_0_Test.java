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
        ListingProductDetails product = new ListingProductDetails();
        String expectedOutput = "ASIN: XYZ123456789, Availability: 100%, Condition Type: Normal, End Date: 2023-10-01, Featured Category: Electronics, Exchange ID: XYZ123456789, Offer Type: Regular, Ex ID: XYZ123456789, Offer Price: 99.99, Exchange Quantity: 10, Quantity Allocated: 5, Seller Country: USA, Seller Id: XYZ123456789, Seller Nickname: John Doe, Seller Rating: 5.0, Seller State: New York, Start Date: 2023-10-01, Status: Available, Title: XYZ Product Details";
        assertEquals(expectedOutput, product.toString());
    }
}
