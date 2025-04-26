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

class A4j_ActorSearch_3_0_Test {

    private A4j a4j;

    private Search searchMock;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        searchMock = mock(Search.class);
        // Use reflection to set the private Search instance in A4j to our mock
        try {
            java.lang.reflect.Field field = A4j.class.getDeclaredField("search");
            field.setAccessible(true);
            field.set(a4j, searchMock);
        } catch (Exception e) {
            fail("Failed to set up the mock Search object: " + e.getMessage());
        }
    }

    @Test
    void testActorSearch_ValidInputs_ReturnsProductInfo() {
        // Arrange
        String actorName = "Leonardo DiCaprio";
        String mode = "full";
        String page = "1";
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ActorSearch(actorName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ActorSearch(actorName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).ActorSearch(actorName, mode, page);
    }

    @Test
    void testActorSearch_EmptyActorName_ReturnsNull() {
        // Arrange
        String actorName = "";
        String mode = "full";
        String page = "1";
        when(searchMock.ActorSearch(actorName, mode, page)).thenReturn(null);
        // Act
        ProductInfo result = a4j.ActorSearch(actorName, mode, page);
        // Assert
        assertNull(result);
        verify(searchMock).ActorSearch(actorName, mode, page);
    }

    @Test
    void testActorSearch_NullMode_ReturnsProductInfo() {
        // Arrange
        String actorName = "Brad Pitt";
        String mode = null;
        String page = "1";
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ActorSearch(actorName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ActorSearch(actorName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).ActorSearch(actorName, mode, page);
    }

    @Test
    void testActorSearch_InvalidPage_ReturnsProductInfo() {
        // Arrange
        String actorName = "Tom Hanks";
        String mode = "partial";
        String page = "invalid";
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ActorSearch(actorName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ActorSearch(actorName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).ActorSearch(actorName, mode, page);
    }
}
