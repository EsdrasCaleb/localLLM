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

class A4j_ActorSearch_3_4_Test {

    @Test
    public void testActorSearch() {
        // Arrange
        Search search = mock(Search.class);
        ProductInfo expectedProductInfo = mock(ProductInfo.class);
        // Act
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.ActorSearch("John Doe", "Movie", "1");
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search).ActorSearch("John Doe", "Movie", "1");
    }
}
