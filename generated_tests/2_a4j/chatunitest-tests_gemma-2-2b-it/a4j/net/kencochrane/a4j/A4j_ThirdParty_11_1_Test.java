package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

class A4j_ThirdParty_11_1_Test {

    @Test
    void ThirdPartyTest() {
        A4j a4j = new A4j();
        assertEquals(a4j.ThirdParty("sellerId", "type", "page", "status"), new Search().ThirdParty("sellerId", "type", "page", "status"));
    }
}
