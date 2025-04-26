package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfileDetails_toString_14_0_Test {

    @Test
    public void testToString() throws Exception {
        // Arrange
        SellerProfileDetails sellerProfileDetails = Mockito.mock(SellerProfileDetails.class);
        Mockito.when(sellerProfileDetails.getSellerNickname()).thenReturn("Seller Nickname");
        Mockito.when(sellerProfileDetails.getOverallFeedbackRating()).thenReturn("5");
        Mockito.when(sellerProfileDetails.getNumberOfFeedback()).thenReturn("10");
        Mockito.when(sellerProfileDetails.getNumberofCanceledAuctions()).thenReturn("2");
        Mockito.when(sellerProfileDetails.getStoreId()).thenReturn("Store ID");
        Mockito.when(sellerProfileDetails.getStoreName()).thenReturn("Store Name");
        Mockito.when(sellerProfileDetails.getSellerFeedBack()).thenReturn(new SellerFeedback());
        // Act
        String result = sellerProfileDetails.toString();
        // Assert
        assertNotNull(result);
        assertEquals("NickName = Seller Nickname\nOverallRating = 5\n# of feedbacks = 10\n# of Canceled Auctions = 2\nStoreId = Store ID\nStoreName = Store Name\nFeedBack = \nSellerFeedback{feedback=null, rating=5, sellerNickname=Seller Nickname, numberOfFeedback=10, numberOfCanceledAuctions=2, storeId=Store ID, storeName=Store Name}\n", result);
    }
}
