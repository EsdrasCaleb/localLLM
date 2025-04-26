package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfileDetails_toString_14_4_Test {

    @Test
    public void testToString() {
        SellerProfileDetails sellerProfileDetails = new SellerProfileDetails();
        sellerProfileDetails.setSellerNickname("John Doe");
        sellerProfileDetails.setOverallFeedbackRating("4.5");
        sellerProfileDetails.setNumberOfFeedback("10");
        sellerProfileDetails.setNumberofCanceledAuctions("5");
        sellerProfileDetails.setStoreId("12345");
        sellerProfileDetails.setStoreName("ABC Store");
        sellerProfileDetails.setSellerFeedBack(new SellerFeedback());
        assertNotNull(sellerProfileDetails.toString());
    }
}
