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

class A4j_BlendedSearch_1_1_Test {

    @Test
    void testBlendedSearch() {
        // Arrange
        String searchTerm = "test";
        String type = "type";
        A4j a4j = new A4j();
        Search search = Mockito.mock(Search.class);
        Mockito.when(search.Blended(searchTerm, type)).thenReturn(new BlendedSearch());
        // Act
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        // Assert
        assertNotNull(result);
        Mockito.verify(search).Blended(searchTerm, type);
    }
}
