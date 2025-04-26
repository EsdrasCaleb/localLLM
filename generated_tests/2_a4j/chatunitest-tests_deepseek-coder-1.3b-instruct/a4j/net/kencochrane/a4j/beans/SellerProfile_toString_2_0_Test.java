package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_0_Test {

    @Test
    public void testToString() {
        SellerProfile sellerProfile = new SellerProfile();
        SellerProfileDetails sellerProfileDetailsMock = mock(SellerProfileDetails.class);
        when(sellerProfileDetailsMock.toString()).thenReturn("Test SellerProfileDetails");
        sellerProfile.setSellerProfileDetails(sellerProfileDetailsMock);
        assertEquals("Test SellerProfileDetails\n", sellerProfile.toString());
    }
}
