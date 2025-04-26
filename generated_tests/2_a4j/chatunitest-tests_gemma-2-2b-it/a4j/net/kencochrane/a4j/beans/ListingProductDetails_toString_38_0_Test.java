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
    void testToString() {
        ListingProductDetails listingProductDetails = Mockito.mock(ListingProductDetails.class);
        Mockito.when(listingProductDetails.toString()).thenReturn("Mock String");
        String expectedString = "Mock String";
        String actualString = listingProductDetails.toString();
        assertEquals(expectedString, actualString);
    }
}
