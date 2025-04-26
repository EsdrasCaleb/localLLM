package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfile_toString_2_1_Test {

    @Test
    void testToString() {
        SellerProfile sellerProfile = new SellerProfile();
        String expectedOutput = "SellerProfileDetails []\n";
        assertEquals(expectedOutput, sellerProfile.toString());
    }
}
