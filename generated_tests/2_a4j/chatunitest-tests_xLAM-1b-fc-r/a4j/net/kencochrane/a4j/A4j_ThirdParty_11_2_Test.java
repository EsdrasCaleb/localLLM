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

public class A4j_ThirdParty_11_2_Test {

    @Test
    public void testThirdParty() {
        // Given
        A4j a4j = new A4j();
        String sellerId = "testId";
        String type = "testType";
        String page = "testPage";
        String status = "testStatus";
        Search search = Mockito.mock(Search.class);
        when(search.ThirdParty(sellerId, type, page, status)).thenReturn(new SellerSearch());
        // When
        SellerSearch result = a4j.ThirdParty(sellerId, type, page, status);
        // Then
        assertEquals(new SellerSearch(), result);
    }
}
