package com.densebrain.rif.client;

import java.rmi.RemoteException;
import java.util.Hashtable;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class RIFManagerFactory_getManager_1_0_Test {

    @Test
    void getManager_existingManager() throws RemoteException {
        // Arrange
        String url = "someUrl";
        RIFManager mockManager = mock(RIFManager.class);
        Hashtable<String, RIFManager> mockMap = new Hashtable<>();
        mockMap.put(url, mockManager);
        RIFManagerFactory factory = new RIFManagerFactory();
        // Set the manager map in the factory
        setField(factory, "managerMap", mockMap);
        // Act
        RIFManager result = factory.getManager(url);
        // Assert
        assertSame(mockManager, result);
        // Verify that the managerMap was not modified
        assertEquals(1, mockMap.size());
    }

    @Test
    void getManager_newManager() throws RemoteException {
        // Arrange
        String url = "anotherUrl";
        RIFManagerFactory factory = new RIFManagerFactory();
        RIFManager mockManager = mock(RIFManager.class);
        when(mockManager.toString()).thenReturn("Mock RIFManager");
        when(new RIFManager(anyString())).thenReturn(mockManager);
        // Act
        RIFManager result = factory.getManager(url);
        // Assert
        assertNotNull(result);
        verify(mockManager).toString();
        // Using assertEquals with equals method for correct comparison
        assertEquals(mockManager, result);
        // Verify that the managerMap was updated
        RIFManager retrievedManager = factory.getManager(url);
        assertSame(mockManager, retrievedManager);
    }

    @Test
    void getManager_nullUrl() {
        RIFManagerFactory factory = new RIFManagerFactory();
        assertThrows(NullPointerException.class, () -> {
            try {
                factory.getManager(null);
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
        });
    }

    // Helper method to access private fields using reflection.
    private void setField(Object obj, String fieldName, Object value) {
        try {
            java.lang.reflect.Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(obj, value);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Error accessing field", e);
        }
    }

    // Dummy classes for compilation
    static class RIFManager {

        String url;

        public RIFManager(String url) {
            this.url = url;
        }

        @Override
        public String toString() {
            return "RIFManager{" + "url='" + url + '\'' + '}';
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null || getClass() != obj.getClass())
                return false;
            RIFManager that = (RIFManager) obj;
            return url != null ? url.equals(that.url) : that.url == null;
        }
    }

    static class RIFManagerFactory {

        private Hashtable<String, RIFManager> managerMap = new Hashtable<>();

        public RIFManager getManager(String url) throws RemoteException {
            if (url == null) {
                throw new NullPointerException("URL cannot be null");
            }
            if (managerMap.containsKey(url)) {
                return managerMap.get(url);
            }
            RIFManager manager = new RIFManager(url);
            managerMap.put(url, manager);
            return manager;
        }
    }
}
