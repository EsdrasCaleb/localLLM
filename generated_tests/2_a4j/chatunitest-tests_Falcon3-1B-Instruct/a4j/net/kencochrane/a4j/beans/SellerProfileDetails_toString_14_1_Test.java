package net.kencochrane.a4j.beans;

import org.junit.Test;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfileDetails_toString_14_1_Test {

    @Test
    public void testToString() {
        // Creating a new SellerProfileDetails object
        SellerProfileDetails sellerProfile = new SellerProfileDetails();
        // Invoking the toString() method
        String result = sellerProfile.toString();
        // Assertions to verify the correctness of the output
        assertEquals("NickName = SellerName, OverallRating = 8.5, Number of feedbacks = 5, Number of Canceled Auctions = 1, StoreId = 123456, StoreName = Local Store", result);
    }
}
